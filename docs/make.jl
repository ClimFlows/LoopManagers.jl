using LoopManagers
using Documenter

DocMeta.setdocmeta!(LoopManagers, :DocTestSetup, :(using LoopManagers); recursive=true)

makedocs(;
    modules=[LoopManagers],
    authors="The ClimFlows contributors",
    sitename="LoopManagers.jl",
    format=Documenter.HTML(;
        canonical="https://ClimFlows.github.io/LoopManagers.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ClimFlows/LoopManagers.jl",
    devbranch="main",
)
