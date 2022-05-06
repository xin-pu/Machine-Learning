using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    /// <summary>
    /// </summary>
    public class TukeyLoss : Loss
    {
        public TukeyLoss(Variable[] variables, NDarray x, NDarray y) : base(variables, x, y)
        {
        }

        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            //Todo
            throw new NotImplementedException();
        }
    }
}