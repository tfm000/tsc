from mambapy.mamba import MambaBlock


__all__ = ['get_block']


def get_block(block: str) -> MambaBlock:
    block = block.lower().capitalize()
    valid_blocks: tuple = ('Mamba', )
    if block not in valid_blocks:
        raise ValueError(f"{block} is not a valid block option. " \
                         "Must be one of {valid_blocks}")
    
    return eval(f'{block}Block')

# TODO: this will fail, as we should be using MambaBlock. Also JambaBlock does 
# not exist. could probabily get around this by selecting mamba or attention in 
# the encoder itself, depending on whether you are on the first encoder or 2nd 
# (i.e. we pass 2 args into forward)