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
        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            //Todo
            throw new NotImplementedException();
        }
    }
}