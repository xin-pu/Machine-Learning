using AutoDiff;
using MLNet.Model.Regression;
using MLNet.Utils;
using Numpy;

namespace MLNet.Loss
{
    /// <summary>
    ///     J(la)= 0.5*sigma((y-yp)^2)
    /// </summary>
    public class LSLoss : LossBase
    {
        public LSLoss(
            Variable[] w,
            NDarray x,
            NDarray y) : base("LSLoss", w, x, y)
        {
        }

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
            switch (Constraint)
            {
                case Constraint.None:
                    return getLeastSquareLoss(variables, x, y);
                case Constraint.L1:
                    return getLeastSquareLoss(variables, x, y) + Lamdba * getLassoLoss(variables);
                case Constraint.L2:
                    return getLeastSquareLoss(variables, x, y) + Lamdba * getRidgeLoss(variables) / 2;
                case Constraint.LP:
                    return getLeastSquareLoss(variables, x, y) + (1 - Lamdba) * getLassoLoss(variables) +
                           Lamdba * getRidgeLoss(variables);
                default:
                    throw new ArgumentOutOfRangeException();
            }
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
            Features = x.shape[1];
            var list = Enumerable.Range(0, batchsize).Select(i =>
            {
                var rowX = x[$"{i}:{i + 1},:"];
                var yp = y[$"{i},:"].GetData<double>();
                var xp = term.matmul(rowX, w);
                var delta = xp - yp[0];

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