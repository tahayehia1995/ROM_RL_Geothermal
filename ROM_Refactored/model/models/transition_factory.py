"""
Factory for creating transition models based on configuration
Note: Only linear transition model is supported. FNO has been removed.
"""

from .linear_transition import LinearTransitionModel, LinearMultiTransitionModel


def create_transition_model(config):
    """
    Create transition model based on configuration.
    Only linear transition model is supported (FNO removed).
    
    Args:
        config: Configuration object with transition type
        
    Returns:
        LinearTransitionModel instance
    """
    transition_type = config['transition'].get('type', 'linear').lower()
    
    # Only support linear transition model
    if transition_type != 'linear':
        import warnings
        warnings.warn(
            f"⚠️ Transition type '{transition_type}' is not supported. "
            f"FNO and hybrid_fno have been removed. Using 'linear' transition model.",
            UserWarning
        )
        transition_type = 'linear'
    
    return LinearTransitionModel(config)

