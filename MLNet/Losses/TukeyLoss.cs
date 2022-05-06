using AutoDiff;
using Numpy;

namespace MLNet.Losses
{
    /// <summary>
    /// </summary>
    public class TukeyLoss : Loss
    {
        internal override Term createLoss(Variable[] w, NDarray x, NDarray y)
        {
            // Todo
            throw new NotImplementedException();
        }
    }
}