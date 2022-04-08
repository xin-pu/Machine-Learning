using MLNet.Utils;
using Xunit;
using Xunit.Abstractions;

namespace MLNetTest.UtilsTest
{
    public class PrimaryFuncTest : AbstractUnitTest
    {
        public PrimaryFuncTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestPolyPrimary()
        {
            var res = PrimaryFunc.getPolyPrimary(2, 4);
            Print(res);
        }

        [Fact]
        public void TestTrigPrimary()
        {
            var res = PrimaryFunc.getTrigPrimary(2, 4);
            Print(res);
        }
    }
}