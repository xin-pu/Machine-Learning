using MLNet.Model.Regression.LinearRegression;
using Numpy;

namespace MLNet.Model.Regression
{
    /// <summary>
    ///     PolynomialFeatures
    ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
    /// </summary>
    public class PolynomialFeatures : MultipleLinearRegression
    {
        public PolynomialFeatures(
            int degree = 1)
            : base("PolynomialFeatures")
        {
            Degree = degree;
        }

        public int Degree { set; get; }


        /// <summary>
        ///     x => 1,x,x^2,x^3,...,x^N
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal override NDarray convert(NDarray x)
        {
            var batch = x.shape[0];
            var features = x.shape[1];
            if (features != 1) throw new Exception("Regression for 1 dims");

            var xTranspose = np.transpose(x);
            var npX = np.ones(Degree + 1, batch);
            Enumerable.Range(1, Degree).ToList().ForEach(d =>
            {
                var row = np.ones(x.shape[0]) * d;
                npX[d] = np.power(xTranspose, row);
            });
            npX = np.transpose(npX);
            return npX;
        }
    }
}