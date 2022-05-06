namespace MLNet.Optimizers
{
    public abstract class Optimizer
    {
    }

    public struct OptimizePara
    {
        public int BatchSize { set; get; }
        public double LearningRate { set; get; }
    }
}