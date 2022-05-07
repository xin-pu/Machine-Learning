using AutoDiff;
using MLNet.Models.Regression;
using MLNet.Utils;
using Numpy;

namespace MLNet.Losses
{
    /// <summary>
    ///     J(la)= 0.5*sigma((y-yp)^2)
    /// </summary>
    public class LSLoss : Loss
    {
        public Constraint Constraint { set; get; }

        public int Features { set; get; }

        public double Lamdba { set; get; } = 0.1;

        /// <summary>
        /// </summary>
        /// <param name="w"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        internal override Term createLoss(Variable[] variables, NDarray x, NDarray y)
        {
            return Constraint switch
            {
                Constraint.None => getLeastSquareLoss(variables, x, y),
                Constraint.L1 => getLeastSquareLoss(variables, x, y) + Lamdba * getLassoLoss(variables),
                Constraint.L2 => getLeastSquareLoss(variables, x, y) + Lamdba * getRidgeLoss(variables) / 2,
                Constraint.LP => getLeastSquareLoss(variables, x, y) + (1 - Lamdba) * getLassoLoss(variables) +
                                 Lamdba * getRidgeLoss(variables),
                _ => throw new ArgumentOutOfRangeException()
            };
        }

        /// <summary>
        ///     训练样本拟合程度
        /// </summary>
        /// <param name="w"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        private Term getLeastSquareLoss(Variable[] w, NDarray x, NDarray y)
        {
            var batchsize = x.shape[0];
            var list = Enumerable.Range(0, batchsize).Select(i =>
            {
                var rowX = x[$"{i}:{i + 1},:"];
                var y_true = y[$"{i},:"].GetData<double>();
                var y_pred_term = term.matmulRow(rowX, w);
                var delta = y_pred_term - y_true[0];

                return TermBuilder.Power(delta, 2);
            });
            return TermBuilder.Sum(list) / batchsize;
        }

        /// <summary>
        ///     防止过拟合 岭回归部分 L2约束
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        private Term getRidgeLoss(Variable[] w)
        {
            var sum = TermBuilder.Sum(w.Select(a => TermBuilder.Power(a, 2)));
            var Ter = TermBuilder.Power(sum, 0.5);
            return Ter;
        }

        /// <summary>
        ///     防止过拟合 岭回归部分 L2约束
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        private Term getLassoLoss(Variable[] w)
        {
            var abs = w.Select(i => TermBuilder.Power(TermBuilder.Power(i, 2), 0.5));
            var sum = TermBuilder.Sum(abs);
            return sum;
        }
    }
}