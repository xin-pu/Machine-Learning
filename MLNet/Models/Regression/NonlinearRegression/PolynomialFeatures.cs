using MLNet.Utils;
using Numpy;

namespace MLNet.Models.Regression
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
        internal override NDarray transform(NDarray x)
        {
            return transformer.to_poly(x, Degree);
        }
    }
}