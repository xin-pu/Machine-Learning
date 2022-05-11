using AutoDiff;
using MLNet.Utils;
using Numpy;

namespace MLNet.Losses
{
    /// <summary>
    ///     逻辑回归的Softmax
    /// </summary>
    public class Softmax : Loss
    {
        public int Features { set; get; }

        internal override Term createLoss(Variable[] variables, NDarray x, NDarray y)
        {
            var batchsize = x.shape[0];
            Features = x.shape[1];

            var list = Enumerable.Range(0, batchsize).Select(i =>
            {
                var rowX = x[$"{i}:{i + 1},:"];
                var yp = y[$"{i},:"].GetData<double>();
                var xp = term.matmulRow(rowX, variables);
                var costSingle = TermBuilder.Log(1 + TermBuilder.Exp(-yp[0] * xp));
                return costSingle;
            });
            return TermBuilder.Sum(list) / batchsize;
        }
    }
}