using System.Text;

namespace MLNet.LearningModel
{
    public struct TrainConfig
    {
        public TrainConfig(
            int epoch = 100,
            int batch = 0,
            double learningRate = 1E-4,
            bool shuffle = true)
        {
            Epoch = epoch;
            Batch = batch;
            LearningRate = learningRate;
            Shuffle = shuffle;
        }

        public int Epoch { get; set; }
        public int Batch { set; get; }
        public double LearningRate { set; get; }
        public bool Shuffle { set; get; }

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"Epoch:\t{Epoch}\rBatch:\t{Batch}\rLR:\t{LearningRate}");
            return str.ToString();
        }
    }
}