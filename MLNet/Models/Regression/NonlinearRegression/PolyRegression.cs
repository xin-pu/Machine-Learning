using MLNet.Transforms;
using Numpy;
using Numpy.Models;

namespace MLNet.Models.Regression
{
    public class PolyRegression : SupervisedModel
    {
        /// <summary>
        ///     非线性回归，多项式函数逼近
        ///     PolynomialFeatures
        ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
        /// </summary>
        /// <param name="degree"></param>
        public PolyRegression(
            int degree = 1)
        {
            Degree = degree;
            Transform = new Polynomial(Degree);
        }

        public int Degree { set; get; }


        internal override NDarray call(NDarray x, Shape shape)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            var y_pred = np.matmul(x, Resolve);
            return np.reshape(y_pred, shape);
        }
    }
}