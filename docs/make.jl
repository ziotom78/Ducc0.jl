using Ducc0
using Documenter

DocMeta.setdocmeta!(Ducc0, :DocTestSetup, :(using Ducc0); recursive=true)

makedocs(;
    modules=[Ducc0],
    authors="Martin Reinecke <martin@MPA-Garching.MPG.DE>, Maurizio Tomasi <ziotom78@gmail.com>",
    repo="https://github.com/ziotom78/Ducc0.jl/blob/{commit}{path}#{line}",
    sitename="Ducc0.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ziotom78.github.io/Ducc0.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ziotom78/Ducc0.jl",
    devbranch="master",
)
