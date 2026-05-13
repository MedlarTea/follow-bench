from irsim.world import ObjectBase


class TargetOmni(ObjectBase):
    def __init__(
        self, color="orange", **kwargs
    ):
        super(TargetOmni, self).__init__(
            color=color, role="target", **kwargs
        )
