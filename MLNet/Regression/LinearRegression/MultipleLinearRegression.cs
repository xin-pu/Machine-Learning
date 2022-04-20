using AutoDiff;
using MLNet.LearningModel;
using MLNet.Loss;
using MLNet.Utils;
using Numpy;

namespace MLNet.Regression.LinearRegression
{
    /// <summary>
    ///     多元线性回归
    ///     y=α + β1*x1 + β2*x2 + ... + βn*xn
    /// </summary>
    public class MultipleLinearRegression : Model
    {
        public MultipleLinearRegression()
            : base("MultipleLinearRegression")
        {
        }

        public override LossBase? CostFunc { get; set; }
        public override NDarray? Resolve { get; set; }

        public NDarray Slove(NDarray x, NDarray y)
        {
            return np.linalg.pinv(x).dot(y);
        }


        internal NDarray slove(NDarray x, NDarray y)
        {
            return np.linalg.pinv(x).dot(y);
        }

        /// <summary>
        ///     x => 1,x1,x2,x3,...,xN
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        internal override NDarray convert(NDarray x)
        {
            return np2.linear_first_order(x);
        }

        internal override void fit(NDarray x, NDarray y, double learning_rate, int epoch)
        {
            var resolve = np.random.randn(x.shape[1], 1);

            var w = new[] {new Variable(), new Variable()};

            CostFunc = new LMSLoss(w, x, y);

            Enumerable.Range(0, 100).ToList().ForEach(e =>
            {
                var theda = resolve?.GetData<double>();

                var loss = CostFunc.CostFunc.Evaluate(w, theda);
                var gradarray = CostFunc.CostFunc.Differentiate(w, theda);
                var grad = np.expand_dims(np.array(gradarray), -1);
                resolve -= learning_rate * grad;

                Log.print?.Invoke($"{Name} Epoch:\t{e:D5}\tLoss:{loss:F4}\r\n");
            });
            Resolve = resolve;
        }

        internal override NDarray call(NDarray x)
        {
            if (Resolve == null) throw new Exception("Resolve is Empty");

            return np.matmul(x, Resolve);
        }
    }
}