"""Build an offline replay viewer and visual review sheets from recorded states."""
import argparse
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COLORS=[(82,174,255),(255,119,107)]
FONT=Path('C:/Windows/Fonts/segoeui.ttf')


def sheet(data, times, path):
    font=ImageFont.truetype(str(FONT),18)
    small=ImageFont.truetype(str(FONT),15)
    canvas=Image.new('RGB',(1260,1000),(15,22,32));text=ImageDraw.Draw(canvas)
    s=data['summary']
    text.text((20,15),f"{s['opponent'].upper()} | seed {s['seed']} | lado invertido: {s['swapped']}",font=font,fill='white')
    text.text((20,45),'Azul: generalista | Vermelho: especialista | Area vermelha: zona de dano | Trilhas: ultimos 10 s',font=small,fill=(188,200,215))
    for k,t in enumerate(times):
        index=min(range(len(data['frames'])),key=lambda i:abs(data['frames'][i]['t']-t));f=data['frames'][index]
        x0=20+(k%3)*415;y0=85+(k//3)*450
        board=Image.new('RGB',(400,400));d=ImageDraw.Draw(board)
        for x in range(100):
            for y in range(100):
                color=(101,109,112) if data['grid'][x][y] else (58,46,31) if data['mud'][x][y] else (29,54,40)
                d.rectangle((4*x,4*y,4*x+3,4*y+3),fill=color)
        layer=Image.new('RGBA',board.size);z=ImageDraw.Draw(layer);edge=f['t']*.25*4
        z.rectangle((0,0,400,edge),fill=(235,55,55,85));z.rectangle((0,400-edge,400,400),fill=(235,55,55,85))
        z.rectangle((0,edge,edge,400-edge),fill=(235,55,55,85));z.rectangle((400-edge,edge,400,400-edge),fill=(235,55,55,85))
        z.rectangle((edge,edge,400-edge,400-edge),outline=(255,125,104,230),width=1)
        board=Image.alpha_composite(board.convert('RGBA'),layer);d=ImageDraw.Draw(board)
        for j in (0,1):
            trail=[(q['p'][j][0]*4,q['p'][j][1]*4) for q in data['frames'][max(0,index-50):index+1]]
            for a,b in zip(trail,trail[1:]):
                if math.dist(a,b)<20:d.line((a,b),fill=COLORS[j],width=1)
        for x,y,owner in f['b']:d.ellipse((x*4-1,y*4-1,x*4+1,y*4+1),fill=COLORS[owner])
        for j,p in enumerate(f['p']):
            x,y=p[0]*4,p[1]*4
            d.ellipse((x-5,y-5,x+5,y+5),fill=COLORS[j],outline='white',width=1)
            d.text((x+7,y-9),str(p[2]),font=small,fill='white',stroke_width=1,stroke_fill='black')
        canvas.paste(board.convert('RGB'),(x0,y0))
        text.text((x0,y0+402),f"{f['t']:.1f}s | vidas {f['p'][0][2]} x {f['p'][1][2]}",font=font,fill='white')
        text.text((x0,y0+425),f"fase: {f['phase'] or 'normal'} | retorno: {f['returning']}",font=small,fill=(188,200,215))
    canvas.save(path)


HTML='''<!doctype html><meta charset="utf-8"><title>Revisão das partidas</title>
<style>body{background:#101720;color:#eaf0f6;font:16px system-ui;margin:24px}h1{font-size:25px}select,button,input{font:inherit;padding:8px;background:#223040;color:white;border:1px solid #456;border-radius:6px}main{display:flex;gap:24px;flex-wrap:wrap}canvas{width:min(640px,90vw);height:min(640px,90vw);background:#16271f}aside{max-width:520px}#events{max-height:440px;overflow:auto}#events button{display:block;text-align:left;width:100%;margin:5px 0;font-size:13px}#time{width:min(500px,70vw)}small{color:#bbc9d5}.stats{line-height:1.7}</style>
<h1>Derrotas primeiro · empates em segundo plano</h1><p>Azul: generalista · vermelho: especialista. Replays dos estados reais, a cada 0,2 s. Nenhuma mudança na política.</p>
<p><select id="match"></select></p><main><section><canvas id="board" width="640" height="640"></canvas><p><button id="play">Reproduzir</button> <select id="speed"><option>1</option><option selected>4</option><option>8</option></select> × <label><input type="checkbox" id="zoom"> Aproximar</label></p><input id="time" type="range" min="0" max="120" step="0.2" value="0"><span id="clock"></span></section><aside><div id="stats" class="stats"></div><h3>Eventos — clique para rever</h3><div id="events"></div></aside></main>
<script>const games=DATA;const sel=document.querySelector('#match'),slider=document.querySelector('#time'),ctx=document.querySelector('canvas').getContext('2d');let game,base,playing=false,last=0,playhead=0;
games.forEach((g,i)=>{const s=g.summary,o=document.createElement('option');o.value=i;o.textContent=`${s.winner===1?'Derrota':'Empate'} | ${s.opponent} | ${s.seed} | ${s.swapped?'invertido':'normal'} | ${s.lives.join(' × ')}`;sel.append(o)});
function choose(){game=games[+sel.value];base=document.createElement('canvas');base.width=base.height=100;let c=base.getContext('2d');for(let x=0;x<100;x++)for(let y=0;y<100;y++){c.fillStyle=game.grid[x][y]?'#656d70':game.mud[x][y]?'#3a2e1f':'#1d3628';c.fillRect(x,y,1,1)}slider.value=playhead=0;document.querySelector('#events').replaceChildren();game.summary.events.forEach(e=>{let b=document.createElement('button');b.textContent=`${e.t.toFixed(2)}s · ${e.event.join(' ')} · vidas ${e.lives.join(' × ')}`;b.onclick=()=>{slider.value=playhead=Math.max(0,e.t-2);draw()};document.querySelector('#events').append(b)});draw()}
function draw(){let t=+slider.value,idx=game.frames.findIndex(f=>f.t>=t);if(idx<0)idx=game.frames.length-1;let f=game.frames[idx],p=f.p;let xmin=0,ymin=0,span=100;if(document.querySelector('#zoom').checked){span=Math.max(25,Math.abs(p[0][0]-p[1][0])+15,Math.abs(p[0][1]-p[1][1])+15);span=Math.min(span,100);xmin=Math.max(0,Math.min(100-span,(p[0][0]+p[1][0]-span)/2));ymin=Math.max(0,Math.min(100-span,(p[0][1]+p[1][1]-span)/2))}const scale=640/span,X=x=>(x-xmin)*scale,Y=y=>(y-ymin)*scale;ctx.clearRect(0,0,640,640);ctx.imageSmoothingEnabled=false;ctx.drawImage(base,xmin,ymin,span,span,0,0,640,640);let edge=f.t*.25;ctx.fillStyle='#ee444455';ctx.fillRect(X(0),Y(0),100*scale,edge*scale);ctx.fillRect(X(0),Y(100-edge),100*scale,edge*scale);ctx.fillRect(X(0),Y(edge),edge*scale,(100-2*edge)*scale);ctx.fillRect(X(100-edge),Y(edge),edge*scale,(100-2*edge)*scale);ctx.strokeStyle='#ff9077';ctx.lineWidth=1;ctx.strokeRect(X(edge),Y(edge),(100-2*edge)*scale,(100-2*edge)*scale);
const colors=['#52aeff','#ff776b'];for(let j=0;j<2;j++){ctx.strokeStyle=colors[j];ctx.beginPath();let old;game.frames.slice(Math.max(0,idx-50),idx+1).forEach(q=>{let a=q.p[j];if(!old||Math.hypot(a[0]-old[0],a[1]-old[1])>5)ctx.moveTo(X(a[0]),Y(a[1]));else ctx.lineTo(X(a[0]),Y(a[1]));old=a});ctx.stroke()}f.u.forEach(a=>{ctx.fillStyle=a[2]==='LIFE'?'#ff88cc':a[2]==='SPEED'?'#66ffbb':'#ffe37a';ctx.fillRect(X(a[0])-3,Y(a[1])-3,6,6)});f.b.forEach(a=>{ctx.fillStyle=colors[a[2]];ctx.beginPath();ctx.arc(X(a[0]),Y(a[1]),2,0,7);ctx.fill()});p.forEach((a,j)=>{ctx.fillStyle=colors[j];ctx.strokeStyle='white';ctx.beginPath();ctx.arc(X(a[0]),Y(a[1]),Math.max(5,.4*scale),0,7);ctx.fill();ctx.stroke();ctx.font='bold 15px system-ui';ctx.fillText(`${j?'S':'G'} ${a[2]}`,X(a[0])+8,Y(a[1])-8)});document.querySelector('#clock').textContent=f.t.toFixed(1)+'s';document.querySelector('#stats').textContent=`Vidas: ${p[0][2]} × ${p[1][2]} | Distância: ${Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]).toFixed(1)} | Fase: ${f.phase||'normal'} | Retorno: ${f.returning?'sim':'não'} | Saída antecipada: ${f.escape?'sim':'não'}`;}
sel.onchange=choose;slider.oninput=()=>{playhead=+slider.value;draw()};document.querySelector('#zoom').onchange=draw;document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pausar':'Reproduzir'};function step(now){if(playing&&last){playhead=Math.min(120,playhead+(now-last)/1000*(+document.querySelector('#speed').value));slider.value=playhead;draw();if(+slider.value>=120)playing=false}last=now;requestAnimationFrame(step)}choose();requestAnimationFrame(step);</script>'''


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    diagnosis=json.loads((args.folder/'diagnosis.json').read_text());assert diagnosis['complete']
    games=[]
    for r in sorted(diagnosis['matches'],key=lambda r:(r['winner']!=1,r['opponent'],r['seed'])):
        p=args.folder/r['replay'];g=json.loads(p.read_text());g['summary']['metrics'].pop('blocked_movement_seconds',None)
        p.write_text(json.dumps(g),encoding='utf-8');games.append(g)
        r['metrics'].pop('blocked_movement_seconds',None)
    (args.folder/'diagnosis.json').write_text(json.dumps(diagnosis,indent=2),encoding='utf-8')
    (args.folder/'replays.html').write_text(HTML.replace('DATA',json.dumps(games,separators=(',',':'))),encoding='utf-8')
    selected=[('markov',1716643869,False,[0,40,65,85,100,120]),
              ('camper',1782557660,True,[90,105,112,116,118.4,120]),
              ('dijkstra',1851487980,False,[85,100,108,112,116,120]),
              ('markov',1561892226,True,[25,26.5,45,75,105,120]),
              ('markov',1681820021,False,[0,30,60,90,110,120])]
    for name,seed,swap,times in selected:
        g=next(g for g in games if g['summary']['opponent']==name and g['summary']['seed']==seed and g['summary']['swapped']==swap)
        sheet(g,times,args.folder/f'{name}_{seed}_{int(swap)}.png')
    print(json.dumps(dict(replays=len(games),sheets=len(selected),viewer=str(args.folder/'replays.html'))))
