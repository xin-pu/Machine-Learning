using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    public class ReLULoss : Loss
    {
        public ReLULoss(Variable[] variables, NDarray x, NDarray y)
            : base("ReLU", variables, x, y)
        {
        }

        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            throw new NotImplementedException();
        }
    }
}