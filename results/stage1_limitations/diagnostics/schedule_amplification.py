"""How much does one reverse step amplify an error in the predicted noise?

GRDM's Gaussian reverse step (tab_ddpm gaussian_p_mean_variance, eps parameterisation, no x0
clipping) gives x_{t-1} = c1 * x0_hat + c2 * x_t + sigma_t z with
x0_hat = sqrt(1/abar_t) x_t - sqrt(1/abar_t - 1) eps_hat, so
d x_{t-1} / d eps_hat = beta_t / (sqrt(1 - abar_t) * sqrt(alpha_t)).

Also reports how often uniform-t training visits each timestep for each Berka table.

Run:  conda run -n tabsyn python schedule_amplification.py
"""

import numpy as np

import common as C
from rdbdiff.third_party.tab_ddpm.gaussian_multinomial_diffsuion import get_named_beta_schedule

T = 2000
b = get_named_beta_schedule("cosine", T)
a = 1.0 - b
ab = np.cumprod(a)
amp = b / np.sqrt(1.0 - ab) / np.sqrt(a)

# Berka non-dimension row counts (full database, data/berka/preprocessed)
rows = {"trans": 1056320, "order": 6471, "client": 5369, "disp": 5369, "account": 4500,
        "card": 892, "loan": 682}
total = sum(rows.values())
steps, batch = 200_000, 4096
visits = {
    t: {
        "seeds_per_batch": batch * n / total,
        "p_batch_has_none": float(np.exp(batch * np.log1p(-n / total))),
        "visits_per_row_total": steps * batch / total,
        "samples_at_t1999_total": steps * batch * n / total / T,
    }
    for t, n in rows.items()
}
C.dump(
    {
        "per_t": {int(t): {"beta": float(b[t]), "abar": float(ab[t]), "sqrt_abar": float(np.sqrt(ab[t])),
                           "eps_to_x_amplification": float(amp[t])}
                  for t in (1999, 1998, 1997, 1995, 1990, 1980, 1950, 1900, 1600, 1000, 400, 100, 10, 1)},
        "n_steps_beta_ge_0.5": int((b >= 0.5).sum()),
        "berka_uniform_seeding": visits,
    },
    "schedule_amplification.json",
)
