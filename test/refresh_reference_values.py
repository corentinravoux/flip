import json

import test_covariance_reference_values
import test_e2e_density
import test_e2e_joint
import test_e2e_velocity


def refresh_covariance_reference_values():
    ref: dict = {}
    for m in test_covariance_reference_values.model_to_test:
        model, kind = m[0], m[1]
        key = f"{model}:{kind}"
        ref[key] = test_covariance_reference_values.test_covariance_reference_metrics(
            model,
            kind,
            debug_return=True,
        )

    out_main = "test_covariance_reference_values.json"
    with open(out_main, "w") as f:
        json.dump(ref, f, indent=2)

    return ref


def refresh_e2e_reference_values():
    ref = {
        "e2e_density": test_e2e_density.test_e2e_density(debug_return=True),
        "e2e_velocity": test_e2e_velocity.test_e2e_velocity(debug_return=True),
        "e2e_joint": test_e2e_joint.test_e2e_joint(debug_return=True),
    }

    out_main = "test_e2e_reference_values.json"
    with open(out_main, "w") as f:
        json.dump(ref, f, indent=2)

    return ref


if __name__ == "__main__":
    refresh_covariance_reference_values()
    refresh_e2e_reference_values()
