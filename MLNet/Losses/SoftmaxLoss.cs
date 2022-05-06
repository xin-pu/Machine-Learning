using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    public class SoftmaxLoss : Loss
    {
        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            /// Todo
            throw new NotImplementedException();
        }
    }
}