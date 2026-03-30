from s0_tuning import S0Trainer, S0Config
import s0_tuning

def test_import():
    assert hasattr(s0_tuning, "__version__")

def test_config_defaults():
    c = S0Config()
    assert c.n_steps == 20
    assert c.lr == 1e-3
    assert c.alpha is None

def test_config_override():
    c = S0Config(n_steps=50, alpha=0.07)
    assert c.n_steps == 50
    assert c.alpha == 0.07
