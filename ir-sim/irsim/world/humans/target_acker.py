from irsim.world import ObjectBase


class TargetAcker(ObjectBase):
    def __init__(self, color="orange", **kwargs):
        super(TargetAcker, self).__init__( color=color, role="target", **kwargs
        )
