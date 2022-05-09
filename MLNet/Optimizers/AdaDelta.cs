using Numpy;

namespace MLNet.Optimizers
{
    public class AdaDelta : Optimizer
    {
        public AdaDelta(double beta = 0.9) : base(0)
        {
            Beta = 0.9;
        }

        public double Beta { protected set; get; }
        public NDarray AccumulativeVariable { set; get; } = null!;
        public NDarray Chi { set; get; } = null!;


        internal override NDarray call(NDarray weight, NDarray grad, int epoch = 0)
        {
            if (epoch == 0)
            {
                AccumulativeVariable = np.zeros_like(weight);
                Chi = np.zeros_like(weight);
            }

            AccumulativeVariable = Beta * AccumulativeVariable + (1 - Beta) * np.square(grad);

            var d = (Chi + Epsilon) / (AccumulativeVariable + Epsilon);
            var ss = np.sqrt(d);
            var deltaGrad = np.multiply(ss, grad);

            Chi = Beta * Chi + (1 - Beta) * np.square(deltaGrad);

            return weight - deltaGrad;
        }
    }
}