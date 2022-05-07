using Numpy;

namespace MLNet.Optimizers
{
    public abstract class Optimizer
    {
        protected Optimizer(double learning_rate)
        {
            Name = GetType().Name;
            LearningRate = learning_rate;
        }

        public string Name { protected set; get; }
        public double LearningRate { protected set; get; }


        public NDarray Call(NDarray weight, NDarray grad)
        {
            return call(weight, grad);
        }

        internal abstract NDarray call(NDarray weight, NDarray grad);
    }
}