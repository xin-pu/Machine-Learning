using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    public class ReLULoss : Loss
    {
        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            throw new NotImplementedException();
        }
    }
}