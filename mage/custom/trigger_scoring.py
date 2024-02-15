if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
    
from mage_ai.orchestration.triggers.api import trigger_pipeline

@custom
def transform_custom(*args, **kwargs):
    trigger_pipeline(
        'scoring',
        variables={},
        check_status=False,
        error_on_failure=True,
        poll_interval=60,
        poll_timeout=None,
        verbose=True,
    )
