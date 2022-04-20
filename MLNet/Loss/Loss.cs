using AutoDiff;
using MLNet.Utils;
using Numpy;

namespace MLNet.Loss
{
    public abstract class LossBase
    {
        public abstract Term? CostFunc { get; set; }
    }

    public class LMSLoss : LossBase
    {
        public LMSLoss(
            Variable[] w,
            NDarray x,
            NDarray y)
        {
            CostFunc = CreateLoss(w, x, y);
        }

        public sealed override Term? CostFunc { get; set; }

        public int Features { set; get; }

        /// <summary>
        /// </summary>
        /// <param name="w"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        internal Term? CreateLoss(Variable[] w, NDarray x, NDarray y)
        {
            var batchsize = x.shape[0];
            Features = x.shape[1];
            var list = Enumerable.Range(0, batchsize).Select(i =>
            {
                var rowX = x[$"{i}:{i + 1},:"];
                var yp = y[$"{i},:"].GetData<double>();
                var xp = np2.matmul(rowX, w);
                var delta = xp - yp[0];

                return TermBuilder.Power(delta, 2);
            });
            return TermBuilder.Sum(list) / batchsize;
        }
    }
}