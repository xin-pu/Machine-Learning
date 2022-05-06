using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    public class SoftmaxLoss : Loss
    {
        public SoftmaxLoss(Variable[] variables, NDarray x, NDarray y)
            : base("SoftmaxLoss", variables, x, y)
        {
        }

        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            /// Todo
            throw new NotImplementedException();
        }
    }
}