using Numpy;

namespace MLNet.Metrics
{
    /// <summary>
    ///     ConfusionMatrix
    ///     混淆矩阵
    ///     pred==c  |  pred!=c     |
    ///     TP       |      FN      |   true==c
    ///     FP       |      TN      |   true!=c
    /// </summary>
    public class ConfusionMatrixs
    {
        public ConfusionMatrixs(NDarray y_true, NDarray y_pred)
        {
            var classes = y_true.GetData<int>().Distinct();
            ConfusionMatrixDict = classes
                .ToList()
                .OrderBy(c => c)
                .ToDictionary(c => c, c => new ConfusionMatrix(y_true, y_pred, c));
        }


        public Dictionary<int, ConfusionMatrix> ConfusionMatrixDict { set; get; }
    }
}