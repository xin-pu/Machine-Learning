using MLNet.Utils;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.DataProvider
{
    public class DataProvider : AbstractUnitTest
    {
        private readonly string singledata = @"..\..\..\..\DataSet\iris.data";

        public DataProvider(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void IrisDataSet()
        {
            var res = np2.loadClassifyData(singledata);
            print(res.Item1);
            print(res.Item2);
            print(res.Item3);
        }
    }
}