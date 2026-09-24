import numpy as np


def gauss_seidel_load_flow(Ybus, V, P, Q, slack_bus,
                           tolerance=1e-6, max_iterations=100):
    """
    Gauss-Seidel load-flow analysis.

    Parameters:
        Ybus          : Bus admittance matrix
        V             : Initial complex bus voltages
        P             : Specified active power at buses (pu)
        Q             : Specified reactive power at buses (pu)
        slack_bus     : Index of slack bus
        tolerance     : Convergence tolerance
        max_iterations: Maximum number of iterations

    Returns:
        V, iterations
    """

    n = len(V)

    for iteration in range(1, max_iterations + 1):

        V_old = V.copy()

        for i in range(n):

            # Slack bus voltage remains fixed
            if i == slack_bus:
                continue

            sum_yv = 0j

            for j in range(n):
                if j != i:
                    sum_yv += Ybus[i, j] * V[j]

            # Specified complex power
            S = complex(P[i], Q[i])

            # Gauss-Seidel voltage equation
            V[i] = (
                (np.conj(S) / np.conj(V[i])) - sum_yv
            ) / Ybus[i, i]

        # Check convergence
        error = np.max(np.abs(V - V_old))

        if error < tolerance:
            return V, iteration

    return V, max_iterations


def calculate_power(Ybus, V):
    """
    Calculate injected active and reactive power at each bus.
    """

    current = Ybus @ V
    S = V * np.conj(current)

    P = S.real
    Q = S.imag

    return P, Q


def display_results(V, P, Q):
    print("\n" + "=" * 60)
    print("              LOAD FLOW ANALYSIS RESULTS")
    print("=" * 60)

    print(
        f"{'Bus':<8}"
        f"{'Voltage(pu)':<15}"
        f"{'Angle(deg)':<15}"
        f"{'P(pu)':<12}"
        f"{'Q(pu)':<12}"
    )

    print("-" * 60)

    for i in range(len(V)):

        magnitude = abs(V[i])
        angle = np.degrees(np.angle(V[i]))

        print(
            f"{i + 1:<8}"
            f"{magnitude:<15.4f}"
            f"{angle:<15.4f}"
            f"{P[i]:<12.4f}"
            f"{Q[i]:<12.4f}"
        )

    print("=" * 60)


def main():

    print("=" * 60)
    print("          3-BUS LOAD FLOW ANALYSIS")
    print("             GAUSS-SEIDEL METHOD")
    print("=" * 60)

    # --------------------------------------------------
    # Y-BUS MATRIX
    # --------------------------------------------------

    # Example 3-bus system in per-unit
    Ybus = np.array([
        [10 - 20j, -5 + 10j, -5 + 10j],
        [-5 + 10j, 8 - 16j, -3 + 6j],
        [-5 + 10j, -3 + 6j, 8 - 16j]
    ], dtype=complex)

    # --------------------------------------------------
    # BUS DATA
    # --------------------------------------------------

    # Bus 1 = Slack
    # Bus 2 = PQ
    # Bus 3 = PQ

    # Specified active power
    # Positive = generation
    # Negative = load
    P = np.array([
        0.0,
        -0.50,
        -0.60
    ])

    # Specified reactive power
    # Positive = generation
    # Negative = load
    Q = np.array([
        0.0,
        -0.20,
        -0.25
    ])

    # Initial voltage values
    V = np.array([
        1.05 + 0j,
        1.00 + 0j,
        1.00 + 0j
    ], dtype=complex)

    slack_bus = 0

    # --------------------------------------------------
    # LOAD FLOW CALCULATION
    # --------------------------------------------------

    V, iterations = gauss_seidel_load_flow(
        Ybus,
        V,
        P,
        Q,
        slack_bus,
        tolerance=1e-6,
        max_iterations=100
    )

    # Calculate final bus power
    P_calculated, Q_calculated = calculate_power(Ybus, V)

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print(f"\nConverged in {iterations} iterations.")

    display_results(
        V,
        P_calculated,
        Q_calculated
    )


if __name__ == "__main__":
    main()
