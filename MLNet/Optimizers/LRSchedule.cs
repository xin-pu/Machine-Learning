using Numpy;

namespace MLNet.Optimizers
{
    public abstract class LRSchedule : SGD
    {
        protected LRSchedule(double learningrate)
            : base(learningrate)
        {
        }

        internal override NDarray call(NDarray weight, NDarray grad, int epoch)
        {
            WorkLearningRate = UpdateLearningRate(epoch);
            return base.call(weight, grad, epoch);
        }

        internal abstract double UpdateLearningRate(int epoch);
    }
}