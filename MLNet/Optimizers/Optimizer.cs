using Numpy;

namespace MLNet.Optimizers
{
    public abstract class Optimizer
    {
        internal const double Epsilon = 1E-7;

        protected Optimizer(double learning_rate)
        {
            Name = GetType().Name;
            LearningRate = learning_rate;
        }

        public string Name { protected set; get; }
        public double LearningRate { protected set; get; }


        public NDarray Call(NDarray weight, NDarray grad, int epoch)
        {
            return call(weight, grad, epoch);
        }

        internal abstract NDarray call(NDarray weight, NDarray grad, int epoch = 0);
    }
}