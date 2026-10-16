# amspork3 sync env renames (proposed)

"stacked X/s" = `\underset{\textrm{s}}{X}` (as Qtl/qtl already do)

## Rendered
- vert, Vert -> v, V
- edge, Edge -> e, E
- trVerts, trEdges, trNewVerts, trReachable -> trV, trE, trVNew, trReach
- vertKey, vertMask, vertValue -> vEff, vMask, vVis
- NewVert -> stacked V/new
- (new metavariable) -> stacked v/new
- X_s on stacked X/new -> stacked X_s/new (subscript inside stack)
- EnvKey, envKey -> EnvEff, envEff
- EnvValue, envValue -> EnvVis, envVis
- tlSig, TlSig -> ob, Ob
- pend, Pend -> aw, Aw
- TlSig1, TlSig2 -> Ob^in, Ob^out
- Pend1, Pend2 -> Aw^in, Aw^out
- TlSig_{e,v} -> Ob_{e,v}
- Pend_{e,v} -> Aw_{e,v}
- Pend_{free,new} -> stacked Aw/{free,new}
- Pend_await (TrNewAccess arg) -> stacked Aw/new
- Tid_{new,fence,arrive,await,req,all} -> stacked Tid/{new,fence,arrive,await,req,all} (all: textsf -> textrm)
- first set, second set -> inbound set, outbound set
- timeline signature -> observer

## Macros (clash fixes)
- \sporkv, \sporkV -> \sporkvalue, \sporkValue
- \sporke -> \sporkexpr

## Macros (follow rendered)
- \sporktr{Verts,Edges,NewVerts,Reachable} -> \sporktr{V,E,VNew,Reach}
- \sporkvert{Key,Mask,Value} -> \sporkv{Eff,Mask,Vis}
- \sporkNewVert -> \sporkVNew[sub]
- (new) -> \sporkvNew[sub]
- delete "didn't define newVert" comment in amspork3_defs.tex
- \spork{E,e}nvKey -> \spork{E,e}nvEff
- \spork{E,e}nvValue -> \spork{E,e}nvVis
- \spork{t,T}lSig -> \spork{o,O}b
- \sporkTlSig{A,B} -> \sporkOb{In,Out}
- \sporkpend, \sporkPend -> \sporkaw, \sporkAw
- \sporkPend{A,B} -> \sporkAw{In,Out}
- \sporkPend{Await,Free} -> \sporkAw{New,Free}
- \sporkPend_{e,v} -> \sporkAw_{e,v}
- \sporkPend_\textrm{{free,new}} -> \sporkAw{Free,New}
- \sporkTid_\textrm{{new,fence,arrive,await,req}} -> \sporkTid{New,Fence,Arrive,Await,Req}
- \sporkTid_\textsf{all} -> \sporkTidAll
