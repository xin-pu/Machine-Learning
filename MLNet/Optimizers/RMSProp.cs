using Numpy;

namespace MLNet.Optimizers
{
    public class RMSProp : Optimizer
    {
        public RMSProp(double learning_rate, double beta = 0.9)
            : base(learning_rate)
        {
            Beta = beta;
        }


        public double Beta { protected set; get; }

        public NDarray AccumulativeVariable { set; get; } = null!;

        internal override NDarray call(NDarray weight, NDarray grad, int epoch = 0)
        {
            if (epoch == 0)
                AccumulativeVariable = np.zeros_like(weight);

            AccumulativeVariable = Beta * AccumulativeVariable + (1 - Beta) * np.square(grad);

            return UpdateWeight(weight, grad);
        }

        internal NDarray UpdateWeight(NDarray weight, NDarray grad)
        {
            var d = LearningRate / np.sqrt(AccumulativeVariable + Epsilon);
            var delta = np.multiply(d, grad);
            var updateWeight = np.subtract(weight, delta);
            return updateWeight;
        }
    }
}