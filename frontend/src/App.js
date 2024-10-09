import './App.css';
import {useState,useEffect} from 'react';
import Searchbar from './components/searchbar.js'
import PlayerInfo from './components/playerinfo.js'
const backendURL = 'http://127.0.0.1:5000/'

function App() {
  const [players,setPlayers] = useState([])
  const [selectedPlayer, setSelectedPlayer] = useState("")
  const [playerData , setPlayerData] = useState([])

  useEffect ( () => {
    const playerFetch = async () => {
      try{
        const res = await fetch(backendURL)
        // Check if the response is ok (status code 200-299)
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        const data = await res.json()
        setPlayers(data)
      } catch (error) {
        console.error(error)
      }
    }
    playerFetch()
  }, [])

  useEffect ( () => {
    const playerFetch = async() => {
      try{
        const json_data = JSON.stringify({"name" : selectedPlayer})
        const res = await fetch(backendURL + 'getPlayer', {
          method: "POST",
          headers : {"Content-Type" : "application/json",
            "Accept" : 'application/json'
          },
          body: json_data
        })
        const data = await res.json()
        let stats_array = data['stats']
        let i = 0;
        while (i<stats_array.length){
          if (i < stats_array.length-1){
            if (stats_array[i]['age'] == stats_array[i+1]['age']){
              stats_array.splice(i,1)
            }
            else{
              i += 1
            }
          }
          else{
            i+=1
          }
        }
        setPlayerData(data)
      } catch (error){
        console.error(error)
      }
    }
    if (selectedPlayer != ""){
      playerFetch()
    }
  }, [selectedPlayer])

  const handleClick = () => {
    const playerFetch = async (sample) => {
        try{
            const json_data = JSON.stringify({"sample" : sample})
            const res = await fetch(backendURL + 'predict', {
              method: "POST",
              headers : {"Content-Type" : "application/json",
                "Accept" : 'application/json'
              },
              body: json_data
            })
            const data = await res.json()
            setPlayerData((prevData) => ({
              ...prevData,
              stats: [...prevData.stats, data], // Add the new stat to the stats array
            }));
          } catch (error){
            console.error(error)
          }
    }
    let sample = createSample()
    playerFetch(sample)
  }

  const createSample = () => {
      let age = playerData.stats[(playerData.stats.length)-1]['age'] + 1
      let year = Number(playerData.stats[(playerData.stats.length)-1]['season'].substring(0,4)) + 1
      let season_one = playerData.stats.length >= 1 ? formatArray(playerData.stats[(playerData.stats.length)-1]) : Array(11).fill(0)
      let season_two = playerData.stats.length >= 2 ? formatArray(playerData.stats[(playerData.stats.length)-2]) : Array(11).fill(0)
      let sample = playerData.info.concat(year,age,season_one,season_two)
      return sample
  }

  const formatArray = (data) => {
      if (data.length == 0){
          return Array(11).fill(0)
      }
      let ret = [
          data['gp'], data['mp'], data['pts'], data['reb'], data['ast'], data['3p'], data['fg%'], data['ft%'], data['stl'], data['blk'], data['tov']
      ]
      return ret
  }

  return (
    <div className="home">
      <span className = 'center header'>Choose a Player</span>
      <Searchbar players = {players} setSelectedPlayer = {setSelectedPlayer} />
      <PlayerInfo playerData = {playerData} />
      {(!playerData || !playerData.stats) ? (<></>) :  (<span className = "predict" onClick ={handleClick}> Predict </span>)}
      
    </div>
  );
}

export default App;
