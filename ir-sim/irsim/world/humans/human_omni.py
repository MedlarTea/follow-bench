from irsim.world import ObjectBase


class HumanOmni(ObjectBase):
    def __init__(
        self, color="b", **kwargs
    ):
        super(HumanOmni, self).__init__(
            color=color, role="human", **kwargs
        )
