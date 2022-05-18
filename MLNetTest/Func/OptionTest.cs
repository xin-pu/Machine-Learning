using System.Collections.Generic;
using System.Collections.Specialized;
using LaYumba.Functional;
using Xunit;
using Xunit.Abstractions;
using static LaYumba.Functional.F;

namespace MLNetTest.Func
{
    public class OptionTest : AbstractUnitTest
    {
        public OptionTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestOption()
        {
            var res = greet(Some("Xin"));
            print(res);
        }

        [Fact]
        public void TestLookUp()
        {
            var list = new NameValueCollection();


            var a = new List<double> {1, 2, 3};
        }

        private string greet(Option<string> greetee)
        {
            return greetee.Match(
                () => "Sorry?",
                name => $"Hello, {name}");
        }
    }
}