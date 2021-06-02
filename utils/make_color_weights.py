import numpy as np


def main():
    q_prior = np.load('resources/q-prior.npy')
    prior_probs = q_prior

    alpha = 5
    gamma = .5
    # define uniform probability
    uni_probs = np.zeros_like(prior_probs)
    uni_probs[prior_probs != 0] = 1.
    uni_probs = uni_probs / np.sum(uni_probs)

    # convex combination of empirical prior and uniform distribution
    prior_mix = (1 - gamma) * prior_probs + gamma * uni_probs

    # set prior factor
    prior_factor = prior_mix ** -alpha
    prior_factor = prior_factor / np.sum(prior_probs * prior_factor)  # re-normalize

    # implied empirical prior
    implied_prior = prior_probs * prior_factor
    implied_prior = implied_prior / np.sum(implied_prior)  # re-normalize

    np.save('resources/weights.npy', implied_prior)


if __name__ == '__main__':
    main()
