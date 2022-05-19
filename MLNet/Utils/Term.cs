using AutoDiff;
using Numpy;

namespace MLNet.Utils
{
    public class term
    {
        public static Term matmulRow(NDarray x, Variable[] v)
        {
            var features = x.shape[1];

            if (v.Length != features)
                throw new Exception();

            var row = x[$"{0}:"].GetData<double>();
            var allTerm = row.Select((c, i) => c * v[i]);

            return TermBuilder.Sum(allTerm);
        }

        /// <summary>
        ///     防止过拟合 Lasso回归部分 L1范数
        ///     各个元素绝对值之和
        ///     |x1|+|x2|+...+|xn|
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        public static Term getLassoLoss(Variable[] w)
        {
            var abs = w.Select(i => TermBuilder.Power(TermBuilder.Power(i, 2), 0.5));
            var sum = TermBuilder.Sum(abs);
            return sum;
        }

        /// <summary>
        ///     防止过拟合 岭回归部分 L2范数
        ///     各个元素的平方和然后再求平方根
        ///     Sqrt(x1^2+x2^2+...+xn^2)
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        public static Term getRidgeLoss(Variable[] w)
        {
            var sum = TermBuilder.Sum(w.Select(a => TermBuilder.Power(a, 2)));
            var Ter = TermBuilder.Power(sum, 0.5);
            return Ter;
        }

        /// <summary>
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weight"></param>
        /// <returns></returns>
        public static Term sigmoid(Term x, double weight = 1)
        {
            return 1.0 / (TermBuilder.Exp(-weight * x) + 1.0);
        }
    }
}