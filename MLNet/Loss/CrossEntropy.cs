using AutoDiff;
using Numpy;

namespace MLNet.Loss
{
    public class CrossEntropy : LossBase
    {
        public CrossEntropy(string name, Variable[] variables, NDarray x, NDarray y)
            : base(name, variables, x, y)
        {
        }

        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            throw new NotImplementedException();
        }
    }
}