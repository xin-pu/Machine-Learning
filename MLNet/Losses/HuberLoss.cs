using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    /// <summary>
    ///     Huber =   r^2/2       (|r|<=n)
    ///     ----------n|r|-n^2/2  ({r}> n)
    /// </summary>
    public class HuberLoss : Loss
    {
        public HuberLoss(string name, Variable[] variables, NDarray x, NDarray y)
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