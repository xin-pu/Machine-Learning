using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    /// <summary>
    /// </summary>
    public class TukeyLoss : Loss
    {
        public TukeyLoss(string name, Variable[] variables, NDarray x, NDarray y)
            : base(name, variables, x, y)
        {
        }

        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            //Todo
            throw new NotImplementedException();
        }
    }
}