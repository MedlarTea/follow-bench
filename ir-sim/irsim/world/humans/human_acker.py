from irsim.world import ObjectBase


class HumanAcker(ObjectBase):
    def __init__(self, color="b", **kwargs):
        super(HumanAcker, self).__init__( color=color, role="human", **kwargs
        )
