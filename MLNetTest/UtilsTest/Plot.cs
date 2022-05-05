using Xunit;

namespace MLNetTest.UtilsTest
{
    public class Plot
    {
        [Fact]
        public void PlotTest()
        {
            double[] dataX = {1, 2, 3, 4, 5};
            double[] dataY = {1, 4, 9, 16, 25};
            var plt = new ScottPlot.Plot(400, 300);
            plt.AddScatter(dataX, dataY);
            double[] dataX1 = {1, 3, 3, 4, 5};
            double[] dataY2 = {1, 2, 9, 16, 25};
            plt.AddScatter(dataX1, dataY2);
            plt.SaveFig("quickstart.png");
        }
    }
}