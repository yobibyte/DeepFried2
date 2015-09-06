import DeepFried2 as df


class ClassNLLCriterion(df.Criterion):
    """
    Contrary to Torch7, this criterion takes raw probabilities as input and
    relies on Theano's graph-optimizations to generate numerically stable code.

    This might need to change to require log-probabilities when making use of
    cuDNN's softmax in the near future.

    ClassNLLCriterion has two modus operandi, which might be split into two
    criteria in the future:

    1. If the input has the same number of dimensions as the targets:
        - The targets are considered to be probabilities.
        - Computes element-wise cross-entropy, taking the log of the input.
    2. If the input has one more dimension than the targets:
        - The targets are considered to be one-hot encoded class labels.

    This condition might also take the target dtype into account in the future.
    """
    def __init__(self, clip=None, classprob_axis=1):
        """
        - `clip`: if not `None`, clips the incoming probabilites into the range
            [`clip`, 1-`clip`] in order to avoid numerical instabilities of the
            `log` operation. This is not necessary in the 1-hot case.

        - `classprob_axis`: The axis along which the class-probabilities reside,
            i.e. this axis should have the same length as number of classes.
        """
        df.Criterion.__init__(self)
        self.clip = clip
        self.axis = classprob_axis

    def symb_forward(self, symb_input, symb_targets):
        if symb_targets.ndim == symb_input.ndim - 1:
            # 1-hot encoding case.

            if 2 < symb_input.ndim:
                # Need to flatten all "extra" dimensions into the batch-dimension.
                symb_input = df.T.swapaxes(symb_input, 0, self.axis).flatten(2).T
                symb_targets = symb_targets.flatten()

            if self.axis != 1:
                symb_input = df.T.swapaxes(symb_input, 1, self.axis)

            int_targets = df.T.cast(symb_targets, 'int32')
            p_y = symb_input[df.T.arange(symb_targets.shape[0]), int_targets]
            if self.clip is not None:
                p_y = df.T.clip(p_y, self.clip, 1-self.clip)
            return df.T.mean(-df.T.log(p_y))

        elif symb_targets.ndim == symb_input.ndim:
            # This is the case when both are full distributions.
            p_y = symb_input
            if self.clip is not None:
                p_y = df.T.clip(p_y, self.clip, 1-self.clip)
            return df.T.mean(-df.T.sum(symb_targets * df.T.log(p_y), axis=self.axis))

        else:
            assert False, "Mismatch in dimensionalities of `{}` input ({}) and targets ({}).".format(df.typename(self), symb_input.ndim, symb_targets.ndim)
