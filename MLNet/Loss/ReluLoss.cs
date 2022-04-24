using AutoDiff;
using Numpy;

namespace MLNet.Loss
{
    public class ReLULoss : LossBase
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