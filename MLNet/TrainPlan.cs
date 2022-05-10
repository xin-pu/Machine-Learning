using System.Text;

namespace MLNet
{
    public struct TrainPlan
    {
        public TrainPlan(
            int epoch = 100,
            int batch = 0,
            double learningRate = 1E-3,
            bool shuffle = true)
        {
            Epoch = epoch;
            Batch = batch;
            LearningRate = learningRate;
            Shuffle = shuffle;
        }

        public int Epoch { get; set; }
        public int Batch { set; get; }
        public bool Shuffle { set; get; }
        public double LearningRate { set; get; }

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"Epoch:\t{Epoch}\rBatch:\t{Batch}\rLR:{LearningRate}");
            return str.ToString();
        }
    }
}