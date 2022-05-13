using MLNet.Transforms;
using Numpy;
using Numpy.Models;

namespace MLNet.Models.Regression
{
    public class MultipleLinearRegression : SupervisedModel
    {
        /// <summary>
        ///     多元线性回归
        ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
        /// </summary>
        public MultipleLinearRegression()
        {
            Transform = new LinearFirstorder();
        }


        internal override NDarray call(NDarray x, Shape shape)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");
            var y_pred = np.matmul(x, Resolve);
            return np.reshape(y_pred, shape);
        }
    }
}