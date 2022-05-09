using Numpy;

namespace MLNet.Optimizers
{
    /// <summary>
    /// </summary>
    public class AdaGrad : Optimizer
    {
        public AdaGrad(double learning_rate)
            : base(learning_rate)
        {
        }

        public NDarray AccumulativeVariable { set; get; } = null!;

        internal override NDarray call(NDarray weight, NDarray grad, int epoch = 0)
        {
            if (epoch == 0)
                AccumulativeVariable = np.zeros_like(weight);

            AccumulativeVariable += np.square(grad);

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