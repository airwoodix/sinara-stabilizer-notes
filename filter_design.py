import sys
import argparse
import logging
import asyncio
from functools import lru_cache

import sympy as sp
from sympy import pi
import numpy as np
from scipy import signal

from miniconf import Miniconf


class StabilizerParameters:
    Ts = 128 / 100e6  # sampling period
    fs = 1 / Ts  # sampling frequency
    full_scale_code = (1 << 15) - 1  # ADC/DAC full scale
    full_scale_volt = 10.0
    volt_per_lsb = full_scale_volt / full_scale_code

    @classmethod
    def volt_to_mu(cls, val):
        return int(np.round(val / cls.volt_per_lsb))

    @classmethod
    def mu_to_volt(cls, val):
        return val * cls.volt_per_lsb

    @classmethod
    def make_iir_ch_payload(cls, ba, *, y_offset=None, y_min=None, y_max=None):
        return {
            "y_offset": cls.volt_to_mu(y_offset or 0),
            "y_min": cls.volt_to_mu(y_min or -10.0),
            "y_max": cls.volt_to_mu(y_max or 10.0),
            "ba": ba,
        }


def bode(ba, fs, **kwds):
    """
    Numerical Bode diagram (`f`, `mag`, `phase`) for a digital filter
    in Stabilizer format `[b0, b1, b2, -a1, -a2]`.

    Additional keywords are passed to `scipy.signal.freqz`.
    """
    f, h = signal.freqz(ba[:3], [1] + [-a for a in ba[3:]], fs=fs, **kwds)

    mag = 20.0 * np.log10(abs(h))
    phase = np.unwrap(np.angle(h)) * 180.0 / np.pi

    return f, mag, phase


class FilterLibrary:
    """
    Bi-quadratic IIR filters from arXiv:1508.06319 in Stabilizer format.
    """

    # sampling period
    Ts = sp.symbols("Ts")

    # representation parameters (q = 1/z)
    q, s = sp.symbols("q s")

    # filter parameters
    K, g, f0, F0, Q = sp.symbols("K g f0 F0 Q")

    # https://arxiv.org/pdf/1508.06319.pdf (Table III, p. 9)
    library = {
        "LP": K / (1 + s / (2 * pi * f0)),
        "HP": K / (1 + 2 * pi * f0 / s),
        "AP": K * (s / (2 * pi * f0) - 1) / (s / (2 * pi * f0) + 1),
        "I": K * 2 * pi * f0 / s,
        "PI": K * (1 + s / (2 * pi * f0)) / (1 / g + s / (2 * pi * f0)),
        "P": K,
        "PD": K * (1 + s / (2 * pi * f0)) / (1 + s / (2 * pi * f0 * g)),
        "LP2": K / (1 + s / (2 * pi * f0 * Q) + (s / (2 * pi * f0)) ** 2),
        "HP2": K / (1 + 2 * pi * f0 / (s * Q) + (2 * pi * f0 / s) ** 2),
        "NOTCH": K
        * (1 + (s / (2 * pi * f0)) ** 2)
        / (1 + s / (2 * pi * f0 * Q) + (s / (2 * pi * f0)) ** 2),
        "I/HO": K
        / (1 + s / (2 * pi * f0 * g))
        * (2 * pi * f0 / s + 1 / Q + s / (2 * pi * f0)),
    }

    names = list(library.keys())

    @classmethod
    @lru_cache
    def get_ba_sym(cls, name):
        """
        Return [b0, b1, b2, -a1, -a2] in symbolic form for the filter `name`.

        The substitution `F0 = π f0 Ts` is performed in the returned expression.
        """
        H = cls.library[name]

        # Bilinear (Tustin) transform
        Hq = (
            H.subs({cls.s: 2 / cls.Ts * (1 - cls.q) / (1 + cls.q)})
            .subs({pi * cls.f0 * cls.Ts: cls.F0})
            .simplify()
        )

        # split numerator and denominator
        b, a = [expr.expand().collect(cls.q) for expr in sp.fraction(Hq)]

        # extract normalized coefficients
        a = [a.coeff(cls.q, n) for n in range(3)]
        a0 = a[0]
        a = [-expr / a0 for expr in a[1:]]

        b = [b.coeff(cls.q, n) for n in range(3)]
        b = [expr / a0 for expr in b]

        return b + a

    @classmethod
    def get_ba(cls, name, **params):
        """
        Return [b0, b1, b2, -a1, -a2] in numeric form for the filter `name`.

        `params` must contain numerical values for all free variables in the
        target filter's transfer function.
        """
        ba = cls.get_ba_sym(name)
        ba = [sp.S(expr).subs({cls.F0: pi * cls.f0 * cls.Ts}) for expr in ba]

        if "Ts" not in params:
            params["Ts"] = StabilizerParameters.Ts

        # useless but improves the error message if a substitution is missing
        syms = set.union(*(expr.free_symbols for expr in ba))
        p = {s: params[str(s)] for s in syms}

        return [float(expr.evalf(subs=p)) for expr in ba]

    @classmethod
    def bode(cls, name, **params):
        """
        Numerically calculate the Bode diagram (`f`, `mag`, `phase`) of
        the filter `name` with given parameters.
        """
        ba = cls.get_ba(name, **params)
        return bode(ba, fs=1 / params["Ts"])


class ScipyFilter:
    """
    Wrappers around IIR filter design routines in `scipy.signal`
    that output filter designs in Stabilizer format.

    All filter-design functions return a list of Stabilizer-compatible
    biquadratic IIR filter parameters.
    """

    fs = StabilizerParameters.fs

    @classmethod
    def _conv_ba(cls, b, a, K):
        b /= a[0]
        a /= a[0]

        return list(b[:3] * K) + list(-a[1:])

    @classmethod
    def _conv_sos(cls, sos, K):
        # FIXME: gain on last stage
        ix_k = len(sos) - 1
        return [
            cls._conv_ba(ba[:3], ba[3:], 1 if ix != ix_k else K)
            for ix, ba in enumerate(sos)
        ]

    @classmethod
    def iirnotch(cls, f0, Q, K, fs=None):
        b, a = signal.iirnotch(f0, Q, fs or cls.fs)
        return [cls._conv_ba(b, a, K)]

    @classmethod
    def iirpeak(cls, f0, Q, K, fs=None):
        b, a = signal.iirpeak(f0, Q, fs or cls.fs)
        return [cls._conv_ba(b, a, K)]

    @classmethod
    def iirfilter(
        cls, N, Fn, K, rp=None, rs=None, btype="low", ftype="butter", fs=None
    ):
        sos = signal.iirfilter(
            N, Fn, rp, rs, btype, ftype=ftype, output="sos", fs=fs or cls.fs
        )
        return cls._conv_sos(sos, K)


def main():
    parser = argparse.ArgumentParser(
        description="Stabilizer IIR channel configuration tool",
    )
    subparsers = parser.add_subparsers()

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity"
    )
    parser.add_argument(
        "--broker", "-b", default="mqtt", type=str, help="The MQTT broker address"
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="dt/sinara/stabilizer",
        help="The MQTT topic prefix of the target",
    )
    parser.add_argument("--y-offset", type=float, default=0.0, help="Output offset")
    parser.add_argument(
        "--y-max", type=float, default=10.0, help="Maximum output voltage"
    )
    parser.add_argument(
        "--y-min", type=float, default=-10.0, help="Minimum output voltage"
    )
    parser.add_argument(
        "-c",
        "--channel",
        type=int,
        default=0,
        choices=[0, 1],
        help="Stabilizer channel",
    )
    parser.set_defaults(func=lambda _: print("Choose filter source"))

    # FilterLibrary interface
    library_p = subparsers.add_parser("lib")
    library_p.add_argument("name", choices=FilterLibrary.names)
    library_p.add_argument("arguments", nargs="*")
    library_p.add_argument("-a", "--analytical", action="store_true")
    library_p.add_argument("-s", "--show-coeffs", action="store_true")
    library_p.set_defaults(func=ep_library)

    # ScipyFilter interface
    scipy_p = subparsers.add_parser("scipy")
    scipy_p.add_argument("name")
    scipy_p.add_argument("arguments", nargs="*")
    scipy_p.add_argument("-s", "--show-coeffs", action="store_true")
    scipy_p.set_defaults(func=ep_scipy)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.WARN - 10 * args.verbose,
    )
    args.func(args)


def ep_library(args):
    fargs = parse_filter_arguments(args.arguments)
    ba = FilterLibrary.get_ba(args.name, **fargs)

    if args.show_coeffs:
        if args.analytical:
            print(FilterLibrary.get_ba_sym(args.name))
        else:
            print(ba)

    set_iir_settings(args, [ba])


def ep_scipy(args):
    fargs = parse_filter_arguments(args.arguments)
    try:
        sos = getattr(ScipyFilter, args.name)(**fargs)
    except AttributeError:
        sos = ScipyFilter.iirfilter(ftype=args.name, **fargs)

    if args.show_coeffs:
        print(sos)

    set_iir_settings(args, sos)


def set_iir_settings(args, sos):
    # second filter is pass-through if not specified
    if len(sos) < 2:
        sos.append([1, 0, 0, 0, 0])

    if len(sos) > 2:
        sys.exit("Only two biquad IIR filters available")

    payloads = [StabilizerParameters.make_iir_ch_payload(ba) for ba in sos]

    # apply offset and saturation to the last filter
    # FIXME?
    payloads[-1]["y_offset"] = StabilizerParameters.volt_to_mu(args.y_offset)
    payloads[-1]["y_max"] = StabilizerParameters.volt_to_mu(args.y_max)
    payloads[-1]["y_min"] = StabilizerParameters.volt_to_mu(args.y_min)

    async def configure_settings():
        interface = await Miniconf.create(args.prefix, args.broker)

        for n, payload in enumerate(payloads):
            response = await interface.command(f"iir_ch/{args.channel}/{n}", payload)
            print(f"Response: {response}")

    asyncio.run(configure_settings())


# m-labs/artiq: artiq/tools.py:parse_arguments
def parse_filter_arguments(arguments):
    d = {}
    for argument in arguments:
        name, _, value = argument.partition("=")
        try:
            d[name] = float(value)
        except ValueError:
            d[name] = str(value)
    return d


if __name__ == "__main__":
    main()
